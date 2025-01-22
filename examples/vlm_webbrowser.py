from time import sleep

import helium
from dotenv import load_dotenv
from PIL import Image
from selenium import webdriver

from smolagents import CodeAgent, OpenAIServerModel, tool
from smolagents.agents import ActionStep


load_dotenv()
import os


# You could use an open model via an inference provider like Fireworks AI
model = OpenAIServerModel(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    api_base="https://api.fireworks.ai/inference/v1",
    model_id="accounts/fireworks/models/qwen2-vl-72b-instruct",
)

# model = LiteLLMModel(
#     model_id="anthropic/claude-3-5-sonnet-latest",
#     api_key=os.getenv("ANTHROPIC_API_KEY"),
# )

from io import BytesIO

from selenium.common.exceptions import ElementNotInteractableException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


def save_screenshot(step_log: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = step_log.step_number
    if driver is not None:
        for step_logs in agent.logs:  # Remove previous screenshots from logs for lean processing
            if isinstance(step_log, ActionStep) and step_log.step_number <= current_step - 2:
                step_logs.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        step_log.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    step_log.observations = url_info if step_logs.observations is None else step_log.observations + "\n" + url_info
    return


# Initialize driver and agent
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--force-device-scale-factor=1")
chrome_options.add_argument("--window-size=1000,1300")
driver = helium.start_chrome(headless=False, options=chrome_options)


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()


@tool
def close_popups() -> None:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    # Common selectors for modal close buttons and overlay elements
    modal_selectors = [
        "button[class*='close']",
        "[class*='modal']",
        "[class*='modal'] button",
        "[class*='CloseButton']",
        "[aria-label*='close']",
        ".modal-close",
        ".close-modal",
        ".modal .close",
        ".modal-backdrop",
        ".modal-overlay",
        "[class*='overlay']",
    ]

    wait = WebDriverWait(driver, timeout=0.5)

    for selector in modal_selectors:
        try:
            elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector)))

            for element in elements:
                if element.is_displayed():
                    try:
                        # Try clicking with JavaScript as it's more reliable
                        driver.execute_script("arguments[0].click();", element)
                    except ElementNotInteractableException:
                        # If JavaScript click fails, try regular click
                        element.click()

        except TimeoutException:
            continue
        except Exception as e:
            print(f"Error handling selector {selector}: {str(e)}")
            continue
    return "Modals closed"


agent = CodeAgent(
    tools=[go_back, close_popups],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks=[save_screenshot],
    max_steps=20,
    verbosity_level=2,
)

helium_instructions = """
You can use helium to access websites. Don't bother about the helium driver, it's already managed.
First you need to import everything from helium, then you can do other actions!
Code:
```py
from helium import *
go_to('github.com/trending')
```<end_code>

You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>

If it's a link:
Code:
```py
click(Link("Top products"))
```<end_code>

If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.

To scroll up or down, use scroll_down or scrol_up with as an argument the number of pixels to scroll from.
Code:
```py
scroll_down(num_pixels=100000) # This will probably scroll all the way down
```<end_code>
When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
```py
close_popups()
```<end_code>

You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>

Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
Code:
```py
final_answer("YOUR_ANSWER_HERE")
```<end_code>

If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To find elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually!
Of course you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url. Don't kill the browser.
"""
agent.run(
    """
I'm trying to find how hard I have to work to get a repo in github.com/trending.
Can you navigate to the profile for the top author of the top trending repo, and give me their total number of commits over the last year?
"""
    + helium_instructions
)
