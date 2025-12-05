# Task-Aware Semantic Map++: Cost-Efficient Task Assignment with Advanced Benchmark



## Environment Setup
```bash
# create conda environment
conda create -n tasmap_plus_plus python=3.8 -y
conda activate tasmap_plus_plus

# install dependencies
pip install -r requirements.txt
```


## Demo
```bash
# demo for task assignment
python demo_task_assignment.py --prompt_file prompts/multiple_task_role.txt

# demo for open-set task assignment
python demo_task_assignment.py --prompt_file prompts/multiple_task_role_open_set.txt
```