from io import BytesIO
import base64
from PIL import Image
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def send_image_sample_request():
    import requests
    
    img = Image.open("/home/fizainef/FRAD034_C07228_00033.jpg")
    byte_io = BytesIO()
    img.save(byte_io, 'png')
    byte_io.seek(0)
    output = requests.post("https://localhost:443/transcribe/stream", files={"image": byte_io}, stream=True, verify=False)
    for line in output.iter_lines():
        print(line.decode("utf-8"), flush=True)

def send_sample_request():
    import requests
    
    prompt = Image.open("/home/fizainef/FRAD034_C07228_00033.jpg")
    buffered = BytesIO()
    prompt.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    sample_input = {"image": img_str, "stream": True, "max_tokens": 4096}
    output = requests.post("https://localhost:443/transcribe", json=sample_input, stream=True, verify=False)
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
        output = requests.post("http://localhost:443/", json=sample_input)
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
    send_image_sample_request()
    # send_sample_request()