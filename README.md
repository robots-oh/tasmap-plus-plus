# Task-Aware Semantic Map++: Cost-Efficient Task Assignment with Advanced Benchmark


| [**Project Page**](https://robots-oh.github.io/tasmap_pp/) | [**Paper**](https://ieeexplore.ieee.org/abstract/document/11361083) |


[Daewon Choi](https://choibigo.github.io/space/)\*,
[Soeun Hwang](https://sites.google.com/view/robots-oh/members#h.oc8l44mgli6o)\*,
[Yoonseon Oh](https://sites.google.com/view/robots-oh/yoonseon-oh)


![TASMap++ Figure](./assets/tasmap_plus_plus.jpg)

## TODO
- [x] Add demo execution.
- [x] Provide benchmark website.
- [x] Add data collection code by loading scenes.
- [x] Add TASMap++ execution code.
- [ ] Upload the TASMap++ benchmark.
- [ ] Add context-aware grounding code.
<!-- - [ ] Merge Docker containers into a single container. -->

## Updates
This repository provides:
- A **task assignment demo** (closed-set and open-set),
<!-- - A **interactive website** including:
  - an **annotation workflow** originally used for data collection (collection is closed, but a test login is available), 
  - a **benchmark viewer** for browsing benchmarks, and
  - a **TASMap++ result viewer** for browsing TASMap++ results. -->
- Data collection and TASMap++ process.




## Environment Setup
### Quick Demo
```bash
# create conda environment
conda create -n tasmap_plus_plus python=3.8 -y
conda activate tasmap_plus_plus

# install dependencies
pip install -r tasmap_pp/requirements.txt

# quick demo
python tasmap_pp/demo_task_assignment.py
```

### Data Collection
Please see [README](https://github.com/robots-oh/tasmap-plus-plus/tree/main/data_collection) in `data_collection` folder.


### Run TASMap++
Please see [README](https://github.com/robots-oh/tasmap-plus-plus/tree/main/tasmap_pp) in `tasmap_pp` folder.








<!-- ## Interactive Website 

### Quick Links
- 🏠 **Main**: [Open Main Page][main]
- ✍️ **Annotation**: [Test Annotation Workflow][annot]
- ▶️ **Benchmark Viewer**: [Run Benchmark Viewer][bench_viewer]
- 🔎 **Result Viewer**: [Run TASMap++ Result Viewer][result_viewer]

If you are redirected to the [main page][main], click the appropriate button (Start Annotation / View Benchmark / View TASMap++ Results), then resume the process.

### Annotation Workflow
1. Click [Start Annotation][annot]
2. Enter your name and date of birth to log in, then click **Start**.

⚠️ Annotation collection has ended, but you can still test the annotation flow.
Log in with **Test credentials**.
- **Test credentials**: `anonymous` / `20250101`




### Benchmark Viewer
1. Click [View Benchmark][bench_viewer]
2. Select **Scenario → House → Room**
3. Click **Next**




### TASMap++ Result Viewer
1. Click [View TASMap++ Results][bench_viewer]
2. Select **Scenario → House → Room**
3. Click **Next**





[main]: http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/index.html
[bench_viewer]: http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/select.html?mode=bench
[result_viewer]: http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/select.html?mode=results
[annot]: http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/index.html#login -->

  


## Bibtex
```bibtex
@article{choi2026task,
  title={Task-Aware Semantic Map++: Cost-Efficient Task Assignment with Advanced Benchmark},
  author={Choi, Daewon and Hwang, Soeun and Oh, Yoonseon},
  journal={IEEE Robotics and Automation Letters},
  year={2026},
  publisher={IEEE}
}
```