from helium import get_driver
from smolagents import CodeAgent, HfApiModel, LiteLLMModel
from PIL import Image
import tempfile
import helium
from selenium import webdriver

# model = HfApiModel("Qwen/Qwen2-VL-7B-Instruct")
# model = HfApiModel("https://lmqbs8965pj40e01.us-east-1.aws.endpoints.huggingface.cloud")

model = LiteLLMModel("gpt-4o")

def save_screenshot(step_log, agent):
    driver = get_driver()
    if driver is not None:
        for step_logs in agent.logs: # Remove previous screenshots from logs since they'll be replaced now
            step_logs.observations_images = None
        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as tmp:
            driver.save_screenshot(tmp.name)
            with Image.open(tmp.name) as img:
                width, height = img.size
                print(f"Captured a browser screenshot: {width}x{height} pixels")
                step_log.observations_images = [img.copy()]  # Create a copy to ensure it persists, important!

    # Update observations with URL
    url_info = f"Current url: {driver.current_url}"
    if step_log.observations is None:
        step_log.observations = url_info
    else:
        step_log.observations += "\n" + url_info
    return


# Initialize driver and agent
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--force-device-scale-factor=1')
chrome_options.add_argument('--window-size=900,1200')
driver = helium.start_chrome("google.com", headless=False, options=chrome_options)

agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks = [save_screenshot],
    max_steps=15,
)

# Run agent
helium_instructions = """
You can use helium to access websites. Don't bother about the helium driver, it's already managed.
First you need to import everything from helium, then you can do other actions!
Code:
```py
from helium import *
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
To find elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually!
Of course you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url. Don't kill the browser either.
"""
agent.run("""
I'm trying to find if I need to work a lot to be impactful.
Find me the current trending repos on GitHub, navigate to the top one,
find its top contributor and tell me how many commits they did.
""" + helium_instructions)