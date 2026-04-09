import json
from typing import AsyncGenerator

from time import sleep
from fastapi import BackgroundTasks
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

from ray import serve


import time
from processing.preprocessing import prepare_image
from processing.postprocessing import postprocess_line, postprocess_output
from templates.system.HTR import SYSTEM
from templates.tasks.HTR import PROMPT
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from processing.preprocessing import get_inputs
import asyncio
from typing import List


@serve.deployment(ray_actor_options={"num_gpus": 1})
class VLLMPredictDeployment:
    def __init__(self, **kwargs):
        """
        Construct a VLLM deployment.

        Refer to https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py
        for the full list of arguments.

        Args:
            model: name or path of the huggingface model to use
            download_dir: directory to download and load the weights,
                default to the default cache dir of huggingface.
            use_np_weights: save a numpy copy of model weights for
                faster loading. This can increase the disk usage by up to 2x.
            use_dummy_weights: use dummy values for model weights.
            dtype: data type for model weights and activations.
                The "auto" option will use FP16 precision
                for FP32 and FP16 models, and BF16 precision.
                for BF16 models.
            seed: random seed.
            worker_use_ray: use Ray for distributed serving, will be
                automatically set when using more than 1 GPU
            pipeline_parallel_size: number of pipeline stages.
            tensor_parallel_size: number of tensor parallel replicas.
            block_size: token block size.
            swap_space: CPU swap space size (GiB) per GPU.
            gpu_memory_utilization: the percentage of GPU memory to be used for
                the model executor
            max_num_batched_tokens: maximum number of batched tokens per iteration
            max_num_seqs: maximum number of sequences per iteration.
            disable_log_stats: disable logging statistics.
            engine_use_ray: use Ray to start the LLM engine in a separate
                process as the server process.
            disable_log_requests: disable logging requests.
        """
        args = AsyncEngineArgs(**kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def stream_results(self, results_generator, params) -> AsyncGenerator[bytes, None]:
        num_returned = 0
        ret = ""
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            ret += text_output
            if text_output == "\n":
                if not "```" in ret:
                    ret = postprocess_line(ret, *params)
                    yield (json.dumps(ret) + "\n").encode("utf-8")
                ret = ""
            num_returned += len(text_output)
    
    async def stream_multiple_results(self, generators_infos: List, pages_params: List) -> AsyncGenerator[bytes, None]:
        queue = asyncio.Queue()
        async def enqueue_results(generator_info, params):
            async for chunk in self.stream_results(generator_info, params):
                await queue.put(chunk)
        
        tasks = [asyncio.create_task(enqueue_results(gen_info, params)) for gen_info, params in zip(generators_infos, pages_params)]
        async def mark_done():
            await asyncio.gather(*tasks)
            await queue.put(None)  # Sentinel value to indicate completion
        asyncio.create_task(mark_done())
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    async def may_abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    async def __call__(self, request: Request) -> Response:
        """Generate completion for the request.

        The request should be a JSON object with the following fields:
        - prompt: the prompt to use for the generation.
        - stream: whether to stream the results or not.
        - other fields: the sampling parameters (See `SamplingParams` for details).
        """
        request_dict = await request.json()
        
        image = request_dict.pop("image", None)
        
        stream = request_dict.pop("stream", False)
        
        generators_infos = []
        pages_params = []
        requests_ids = []
        if stream:
            background_tasks = BackgroundTasks()
        else:
            outputs_array = []
        sampling_params = SamplingParams(**request_dict)
        
        for inputs, ratio, offset in get_inputs(image, await self.engine.get_tokenizer(), PROMPT, SYSTEM):
            request_id = random_uuid()
            results_generator = self.engine.generate(
                inputs,
                sampling_params,
                request_id=request_id,
            )
            generators_infos.append(results_generator)
            requests_ids.append(request_id)
            pages_params.append((ratio, offset))
            if stream:
                # Using background_taks to abort the the request
                # if the client disconnects.
                background_tasks.add_task(self.may_abort_request, request_id)

        if stream:
            return StreamingResponse(
                self.stream_multiple_results(generators_infos, pages_params), background=background_tasks
            )
        else:
            async def consume_generator(gen, id, params):
                # Non-streaming case
                final_output = None
                async for request_output in gen:
                    if await request.is_disconnected():
                        # Abort the request if the client disconnects.
                        await self.engine.abort(id)
                        return None
                    final_output = request_output
                response = [output.text for output in final_output.outputs]
                outputs = postprocess_output("\n".join(response), *params)
                return outputs

            results = await asyncio.gather(*[consume_generator(gen, id, params) for gen, id, params in zip(generators_infos, requests_ids, pages_params)])
            outputs = []
            for text_outputs in results:
                if text_outputs is None:
                    return Response(status_code=499)
                outputs += text_outputs
            # outputs = postprocess_output("\n".join(outputs))
            return Response(content=json.dumps(outputs))


def send_sample_request():
    import requests
    
    prompt = Image.open("/home/fizainef/FRAD034_C07228_00033.jpg")
    buffered = BytesIO()
    prompt.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    sample_input = {"image": img_str, "stream": True, "max_tokens": 4096}
    output = requests.post("http://localhost:8000/", json=sample_input, stream=True)
    for line in output.iter_lines():
        print(line.decode("utf-8"), flush=True)

def send_multiple_parallel_requests(num_requests: int):
    import requests
    import threading

    def send_request(i: int):
        start_time = time.time()
        prompt = Image.open("/home/fizainef/FRAD034_C07228_00033.jpg")
        buffered = BytesIO()
        prompt.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        sample_input = {"image": img_str, "stream": False, "max_tokens": 4096}
        output = requests.post("http://localhost:8000/", json=sample_input)
        end_time = time.time()
        print(f"Request {i} completed in {end_time - start_time:.2f} seconds")

    threads = []
    for i in range(num_requests):
        thread = threading.Thread(target=send_request, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    # To run this example, you need to install vllm which requires
    # OS: Linux
    # Python: 3.8 or higher
    # CUDA: 11.0 – 11.8
    # GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, etc.)
    # see https://vllm.readthedocs.io/en/latest/getting_started/installation.html
    # for more details.
    
    deployment = VLLMPredictDeployment.bind(model="/home/fizainef/LLM/Weights/Qwen2-5-VL-3B-GRPO-TSV",
                                            max_num_seqs=90,
                                            max_model_len=8192,
                                            max_num_batched_tokens=16384*2,
                                            dtype="bfloat16",
                                            gpu_memory_utilization=0.95,
                                            enable_chunked_prefill=True,
                                            )
    serve.run(deployment)
    start_time = time.time()
    send_sample_request()
    serve.shutdown()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")