import json
from typing import AsyncGenerator, List, Dict, Any, Optional
import asyncio

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
from processing.postprocessing import postprocess_line
from templates.system.HTR import SYSTEM
from templates.tasks.HTR import PROMPT
from PIL import Image
from io import BytesIO
import base64
import numpy as np
from processing.preprocessing import get_inputs


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

    async def process_single_image(
        self, 
        inputs: Any, 
        sampling_params: SamplingParams,
        request_id: str,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Traite une seule image et retourne le résultat.
        
        Args:
            inputs: Les inputs préparés pour le modèle
            sampling_params: Paramètres d'échantillonnage
            request_id: ID unique de la requête
            stream: Si True, retourne un générateur pour le streaming
            
        Returns:
            Dict contenant le texte généré ou un générateur
        """
        results_generator = self.engine.generate(
            inputs,
            sampling_params,
            request_id=request_id,
        )
        
        if stream:
            return {"generator": results_generator, "request_id": request_id}
        
        # Mode non-streaming : attendre le résultat final
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        assert final_output is not None
        text_output = final_output.outputs[0].text
        return {"text": text_output, "prompt": final_output.prompt}

    async def stream_results(self, results_generator) -> AsyncGenerator[bytes, None]:
        """Stream des résultats pour une seule image."""
        num_returned = 0
        ret = ""
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            ret += text_output
            if text_output == "\n":
                if not "```" in ret:
                    ret = postprocess_line(ret)
                    yield (json.dumps(ret) + "\n").encode("utf-8")
                ret = ""
            num_returned += len(text_output)

    async def stream_results_dual(
        self, 
        generator_left, 
        generator_right,
        request_id_left: str,
        request_id_right: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream des résultats pour deux images en parallèle.
        Les résultats sont assemblés : gauche puis droite.
        """
        results_left = []
        results_right = []
        
        async def collect_left():
            num_returned = 0
            ret = ""
            async for request_output in generator_left:
                text_outputs = [output.text for output in request_output.outputs]
                assert len(text_outputs) == 1
                text_output = text_outputs[0][num_returned:]
                ret += text_output
                if text_output == "\n":
                    if not "```" in ret:
                        ret = postprocess_line(ret)
                        results_left.append(ret)
                    ret = ""
                num_returned += len(text_output)
        
        async def collect_right():
            num_returned = 0
            ret = ""
            async for request_output in generator_right:
                text_outputs = [output.text for output in request_output.outputs]
                assert len(text_outputs) == 1
                text_output = text_outputs[0][num_returned:]
                ret += text_output
                if text_output == "\n":
                    if not "```" in ret:
                        ret = postprocess_line(ret)
                        results_right.append(ret)
                    ret = ""
                num_returned += len(text_output)
        
        # Exécuter les deux collectes en parallèle
        await asyncio.gather(collect_left(), collect_right())
        
        # Streamer les résultats assemblés : gauche puis droite
        for result in results_left:
            yield (json.dumps({"page": "left", "content": result}) + "\n").encode("utf-8")
        
        for result in results_right:
            yield (json.dumps({"page": "right", "content": result}) + "\n").encode("utf-8")

    def merge_dual_results(self, result_left: str, result_right: str) -> str:
        """
        Post-traitement pour assembler les résultats de deux pages.
        Par défaut : gauche puis droite, séparés par une ligne vide.
        
        Vous pouvez personnaliser cette fonction selon vos besoins.
        """
        return result_left + "\n\n" + result_right

    async def may_abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    async def __call__(self, request: Request) -> Response:
        """
        Generate completion for the request.

        The request should be a JSON object with the following fields:
        - prompt: the prompt to use for the generation.
        - stream: whether to stream the results or not.
        - other fields: the sampling parameters (See `SamplingParams` for details).
        """
        request_dict = await request.json()
        
        image = request_dict.pop("image", None)
        stream = request_dict.pop("stream", False)
        sampling_params = SamplingParams(**request_dict)
        
        # Récupérer les inputs (peut retourner 1 ou 2 images)
        tokenizer = await self.engine.get_tokenizer()
        inputs_result = get_inputs(image, tokenizer, PROMPT, SYSTEM)
        
        # Vérifier si get_inputs retourne une liste de 2 éléments (double page)
        # ou un seul élément (page simple)
        is_dual_page = isinstance(inputs_result, (list, tuple)) and len(inputs_result) == 2
        
        if is_dual_page:
            # Cas double page : traiter les deux images en parallèle
            inputs_left, inputs_right = inputs_result
            request_id_left = random_uuid()
            request_id_right = random_uuid()
            
            if stream:
                # Mode streaming pour double page
                generator_left = self.engine.generate(
                    inputs_left,
                    sampling_params,
                    request_id=request_id_left,
                )
                generator_right = self.engine.generate(
                    inputs_right,
                    sampling_params,
                    request_id=request_id_right,
                )
                
                background_tasks = BackgroundTasks()
                background_tasks.add_task(self.may_abort_request, request_id_left)
                background_tasks.add_task(self.may_abort_request, request_id_right)
                
                return StreamingResponse(
                    self.stream_results_dual(
                        generator_left, 
                        generator_right,
                        request_id_left,
                        request_id_right
                    ),
                    background=background_tasks
                )
            else:
                # Mode non-streaming pour double page : inférence parallèle
                results = await asyncio.gather(
                    self.process_single_image(inputs_left, sampling_params, request_id_left),
                    self.process_single_image(inputs_right, sampling_params, request_id_right)
                )
                
                result_left = results[0]["text"]
                result_right = results[1]["text"]
                
                # Post-traitement pour assembler les résultats
                merged_text = self.merge_dual_results(result_left, result_right)
                
                ret = {
                    "text": [merged_text],
                    "dual_page": True,
                    "left": result_left,
                    "right": result_right
                }
                return Response(content=json.dumps(ret))
        
        else:
            # Cas page simple : comportement original
            inputs = inputs_result
            request_id = random_uuid()
            results_generator = self.engine.generate(
                inputs,
                sampling_params,
                request_id=request_id,
            )
            
            if stream:
                background_tasks = BackgroundTasks()
                background_tasks.add_task(self.may_abort_request, request_id)
                return StreamingResponse(
                    self.stream_results(results_generator), 
                    background=background_tasks
                )

            # Non-streaming case
            final_output = None
            async for request_output in results_generator:
                if await request.is_disconnected():
                    await self.engine.abort(request_id)
                    return Response(status_code=499)
                final_output = request_output

            assert final_output is not None
            prompt = final_output.prompt
            text_outputs = [prompt + output.text for output in final_output.outputs]
            ret = {"text": text_outputs, "dual_page": False}
            return Response(content=json.dumps(ret))


def send_sample_request():
    import requests
    
    prompt = Image.open("/home/fizainef/FRAD021_2O_041_0040_00008.jpg")
    buffered = BytesIO()
    prompt.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    sample_input = {"image": img_str, "stream": True, "max_tokens": 4096}
    output = requests.post("http://localhost:8000/", json=sample_input)
    for line in output.iter_lines():
        print(line.decode("utf-8"), flush=True)
        sleep(0.5)


def send_multiple_parallel_requests(num_requests: int):
    import requests
    import threading

    def send_request(i: int):
        start_time = time.time()
        prompt = Image.open("/home/fizainef/FRAD021_2O_041_0040_00008.jpg")
        buffered = BytesIO()
        prompt.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        sample_input = {"image": img_str, "stream": False, "max_tokens": 4096}
        output = requests.post("http://localhost:8000/", json=sample_input)
        end_time = time.time()
        print(f"Request {i} completed in {end_time - start_time:.2f} seconds")
        print(f"Response: {output.json()}")

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
    
    deployment = VLLMPredictDeployment.bind(model="/home/fizainef/LLM/Weights/Qwen2-5-VL-3B-GRPO-TSV")
    serve.run(deployment)
    start_time = time.time()
    send_sample_request()
    # send_multiple_parallel_requests(60)
    serve.shutdown()
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")