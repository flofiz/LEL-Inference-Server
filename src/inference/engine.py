import json
import math
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
from processing.postprocessing import postprocess_line, postprocess_output, add_metadata
from templates.system.HTR import SYSTEM
from templates.tasks.HTR import PROMPT
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from processing.preprocessing import get_inputs
import asyncio
from typing import List, Dict

import logging


class HTREngine:
    def __init__(self, name, **kwargs):
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
        self.name = name
        args = AsyncEngineArgs(**kwargs)
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.logger = logging.getLogger("ray.serve")

    async def stream_results(self, results_generator, params) -> AsyncGenerator[bytes, None]:
        num_returned = 0
        ret = ""
        line_perplexities = []
        file_perplexities = []
        char_perplexities = []
        file_perplexity = 0
        line_perplexity = 0
        file_length = 0
        line_length = 0
        char_ppl_count = 0
        char_logprob = 0
        is_char_seq = True
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            logprobs = [output.logprobs for output in request_output.outputs]
            assert len(text_outputs) == 1
            logprob = logprobs[0][-1] if logprobs else None
            text_output = text_outputs[0][num_returned:]
            token_logprob = list(logprob.values())[0].logprob
            # token_perplexity = 2 ** (-token_logprob)
            file_perplexity += token_logprob
            file_length += 1
            ret += text_output
            line_perplexity += token_logprob
            line_length += 1
            if is_char_seq:
                if text_output=="\t":
                    is_char_seq = False
                    char_perplexities.append(math.exp(-line_perplexity/line_length) if line_length>0 else 0)
                else:
                    char_logprob += token_logprob
                    char_ppl_count += 1
            if text_output == "\n":
                if not "```" in ret:
                    line_perplexities.append(math.exp(-line_perplexity/line_length) if line_length>0 else 0)
                    try:
                        ret = postprocess_line(ret, char_perplexity=char_perplexities[-1] if char_perplexities else 0, line_perplexity=line_perplexities[-1] if line_perplexities else 0, *params)
                        yield (json.dumps(ret) + "\n").encode("utf-8")
                    except Exception as e:
                        self.logger.error(f"Erreur lors du traitement de la ligne: {ret}. Détails de l'erreur: {e}", exc_info=True)
                ret = ""
                is_char_seq = True
                line_perplexity = 0
                line_length = 0
                    
            num_returned += len(text_output)
        metadata = {
            "metrics": {
                "file_perplexity": file_perplexity,
                "file_length": file_length,
                "char_perplexity": char_logprob,
                "char_ppl_count": char_ppl_count,
            }
        }
        yield (json.dumps(metadata) + "\n").encode("utf-8")    
    async def stream_multiple_results(self, generators_infos: List, pages_params: List, metadata: Dict) -> AsyncGenerator[bytes, None]:
        queue = asyncio.Queue()
        metadata_json = json.dumps({"metadata": metadata}) + "\n"
        yield metadata_json.encode("utf-8")
        all_metrics = []
        async def enqueue_results(generator_info, params):
            async for chunk in self.stream_results(generator_info, params):
                decoded = json.loads(chunk.decode("utf-8").strip())
                if "metrics" in decoded:
                    all_metrics.append(decoded["metrics"])
                else:
                    await queue.put(chunk)
        
        tasks = [asyncio.create_task(enqueue_results(gen_info, params)) for gen_info, params in zip(generators_infos, pages_params)]
        async def mark_done():
            await asyncio.gather(*tasks)
            if all_metrics:
                avg_file_perplexity = math.exp(-np.sum([m["file_perplexity"] for m in all_metrics])/np.sum([m["file_length"] for m in all_metrics])) if np.sum([m["file_length"] for m in all_metrics])>0 else 0
                avg_char_perplexity = math.exp(-np.sum([m["char_perplexity"] for m in all_metrics])/np.sum([m["char_ppl_count"] for m in all_metrics])) if np.sum([m["char_ppl_count"] for m in all_metrics])>0 else 0
                final_metrics = {
                    "file_perplexity": avg_file_perplexity,
                    "char_perplexity": avg_char_perplexity
                }
                await queue.put((json.dumps({"metadata": {"metrics": final_metrics}}) + "\n").encode("utf-8"))
            await queue.put(None)  # Sentinel value to indicate completion
        asyncio.create_task(mark_done())
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    async def may_abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    async def transcribe(self, request: Request, stream: bool = False) -> Response:
        form = await request.form()
        image = form['image']
        contents = await image.read()
        # request_dict = await request.json()
        self.logger.info(image.filename, exc_info=True)
        request_dict = {"max_tokens": 4096*3, "logprobs": 1}
        image_width, image_height = Image.open(BytesIO(contents)).size
        # image = request_dict.pop("image", None)
        metadata = {"imageWidth": image_width, "imageHeight": image_height, "model": self.name}
        return await self.__call__(request = request, image = contents, prompt = "", metadata = metadata, request_dict = request_dict, stream = stream)

    #TODO: retranscribe endpoint

    async def __call__(self, request: Request, image: Image, prompt: str, metadata: dict, request_dict: dict, stream: bool = False) -> Response:
        """Generate completion for the request.

        The request should be a JSON object with the following fields:
        - prompt: the prompt to use for the generation.
        - stream: whether to stream the results or not.
        - other fields: the sampling parameters (See `SamplingParams` for details).
        """
        
        # stream = request_dict.pop("stream", False)
        request_dict["temperature"] = 0.0
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
                request_id=request_id
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
                self.stream_multiple_results(generators_infos, pages_params, metadata), background=background_tasks
            )
        else:
            #TODO perplexity hors streaming
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
            
            outputs = add_metadata(outputs, image_width=metadata["imageWidth"], image_height=metadata["imageHeight"])
            # outputs = postprocess_output("\n".join(outputs))
            return Response(content=json.dumps(outputs))

