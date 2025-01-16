from helium import get_driver
from smolagents import CodeAgent, HfApiModel, LiteLLMModel
import os
import base64

# model = HfApiModel("Qwen/Qwen2-VL-7B-Instruct")

model = LiteLLMModel("gpt-4o")

def encode_image(abs_path):
    with open(abs_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_screenshot(step_log):
    abs_path = os.path.abspath('screenshot.png')
    driver = get_driver()
    if driver is not None:
        driver.save_screenshot(abs_path)
        base64_image = encode_image(abs_path)
        step_log.observations_images=[base64_image]
        print("SAVED IMAGE", type(base64_image))
    return


agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks = [save_screenshot],
    max_steps=10,
)

helium_instructions = """
You can use helium to access websites:
First you need to import everything from helium, then you can do other actions!
Code:
```py
from helium import *
start_chrome("google.com") # You can optionnaly pass a url to start_chrome to firectly got_to a specific webpage
go_to('github.com/login')
click('mherrmann/helium')
```<end_code>
In general stop your action after each button click to see what happens on your screenshot:
Code:
```py
write('username', into='Username')
write('password', into='Password')
click('Sign in')
```<end_code>
To scroll down, use:
Code:
```py
import time
scroll_down(num_pixels=1000)
time.sleep(0.5)
```<end_code>
Proceed in several steps rather than trying to do it all in one shot.
And at the end, only when you have your answer, return your final answer.
Code:
```py
final_answer("YOUR_ANSWER_HERE")
```<end_code>
You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>
Normally the page loads quickly, no need to wait for many seconds.
To find stuff, just look at the screenshot rather than trying to gather information in code.
Of course you can act on buttons like a user would do when navigating.
Don't try to take screenshots, screenshots will be automatically taken.
"""
agent.run("What's the dataset with the most likes (shown as a heart icon) under hf.co/datasets? YOU HAVE TO SORT THE DATASET by MOST LIKES to get a reliable answer." + helium_instructions)


if os.path.exists("screenshot.png"):
    os.remove("screenshot.png")