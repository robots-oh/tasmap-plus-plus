import os

script_dir = os.path.dirname(os.path.abspath(__file__))


def convert_to_fewshot_messages(file_path, messages):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    entries = [entry.strip() for entry in raw_text.split('---') if entry.strip()]

    # Few-shot 예시 추가
    for entry in entries:
        user_prompt = {
            "role": "user",
            "content": "Find appropriate task for the target.\nTask Candidates: [leave the (target) as it is, relocate the (target), reorient the (target), washing-up the (target), mop the (target), vacuum the (target), wipe the (target), fold the (target), close the (target), turn off the (target), dispose of the (target), empty the (target)].'(target)' should be replaced.\n What is the appropriate task for the target?"
        }
        assistant_response = {
            "role": "assistant",
            "content": entry
        }
        messages.append(user_prompt)
        messages.append(assistant_response)

    return messages


def get_prompts():
    messages = []
    
    # 시스템 프롬프트
    with open(os.path.join(script_dir, 'multiple_task_role.txt'), 'r', encoding='utf-8') as f:
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

    # messages = convert_to_fewshot_messages(os.path.join(script_dir, 'fewshots.txt'), messages)

    # with open(os.path.join(script_dir, 'prompt.txt')) as pt: 
    #     prompt = pt.read()
    prompt = ""
    
    return messages, prompt



def get_messages(prompt_path):
    messages = []
    
    # 시스템 프롬프트
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



def get_prompt_for_response():
    messages = []
    
    with open(os.path.join(script_dir, 'multiple_task_role.txt'), 'r', encoding='utf-8') as f:
        role = f.read()
    system_prompt = {
        "role": "system",
        "content": [
            {
            "type": "input_text",
            "text": role
            }
        ]
    }
    messages.append(system_prompt)
    
    return messages

if __name__=="__main__":
    get_prompts()