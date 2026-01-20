import os
import io
import re
import json
import base64
from PIL import Image
from openai import OpenAI
import random
import yaml
import argparse 
import shutil 

script_dir = os.path.dirname(os.path.abspath(__file__))

try:
    with open("tasmap_pp/configs/demo_config.yaml") as f:
        config = yaml.full_load(f)
    API_KEY = config["openai_api_key"]
except:
    API_KEY = input("Please enter your OpenAI API key: ").strip()
    config = {"openai_api_key": API_KEY}
    with open("tasmap_pp/configs/demo_config.yaml", "w") as f:
        yaml.dump(config, f)

client = OpenAI(api_key=API_KEY)

def get_messages(prompt_path):
    messages = []
    with open(prompt_path, 'r', encoding='utf-8') as f:
        role = f.read()
    system_prompt = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": role
            }
        ]
    }
    messages.append(system_prompt)
    return messages

def assign_tasks(messages, grid_image):
    buffered = io.BytesIO()
    messages = messages.copy()

    grid_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    query = [{
        "role": "user", 
        "content": [
            {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            },
            {
                "type": "text", 
                "text": "I will give you 100 bucks if you respond well. Analyze the image step by step."
            },
        ]
    }]
    messages.extend(query)

    response = client.chat.completions.create(
        model="o4-mini-2025-04-16",
        messages=messages,
        reasoning_effort="high",
        response_format={"type": "text"},
        store=False
    )
    answer = response.choices[0].message.content
    return answer

def parse_respose(response):
    try:
        json_text_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_text_match:
            json_text = json_text_match.group(0)
            data = json.loads(json_text)
            target = data.get("target", "").split('/')[0]
            if isinstance(data.get("task", ""), list):
                task_list = data.get("task", "")
                task = ",".join([item.split()[0] for item in task_list])
            else:
                task = data.get("task", "").split(" ")[0]
            return target, task
    except (json.JSONDecodeError, AttributeError):
        print("Parsing Error !!")
        pass
    
    return "", ""

def choose_from_options(options, name, random_selection=False):
    print("\nHere is a list of available {}s:\n".format(name))

    for k, option in enumerate(options):
        docstring = ": {}".format(options[option]) if isinstance(options, dict) else ""
        print("[{}] {}{}".format(k + 1, option, docstring))
    print()

    if not random_selection:
        try:
            s = input("Choose a {} (enter a number from 1 to {}): ".format(name, len(options)))
            k = min(max(int(s), 1), len(options)) - 1
        except ValueError:
            k = 0
            print("Input is not valid. Use {} by default.".format(list(options)[k]))
    else:
        k = random.choice(range(len(options)))

    return list(options)[k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=f"{script_dir}/prompts/multiple_task_role.txt",
        help="Path to the prompt file."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/demo_scenes",
        help="Path to the data directory."
    )
    args = parser.parse_args()

    prompt_file = args.prompt_file
    data_dir = args.data_dir
    messages = get_messages(prompt_file)

    scene = choose_from_options(os.listdir(data_dir), "Scenario", random_selection=False)
    house = choose_from_options(os.listdir(os.path.join(data_dir, scene)), "House", random_selection=False)
    stitched_images_dir = os.path.join(data_dir, scene, house, "stitched_images")
    results_dir = os.path.join(data_dir, scene, house, "results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)

    files = os.listdir(stitched_images_dir)
    sorted_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    for img_file in sorted_files:

        print("=" * 60)
        print(f"\nProcessing image: {img_file}")
        img_path = os.path.join(stitched_images_dir, img_file)
        grid_image = Image.open(img_path).convert("RGB")

        answer = assign_tasks(messages, grid_image)
        print("\nModel Response:\n", answer)

        target, task = parse_respose(answer)
        grid_image.save(os.path.join(results_dir, f"{os.path.splitext(img_file)[0]}_{target}_{task}.png"))

if __name__ == "__main__":
    main()
