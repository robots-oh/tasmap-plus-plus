# Task-Aware Semantic Map++: Cost-Efficient Task Assignment with Advanced Benchmark


This repository provides:
- A **task assignment demo** (closed-set and open-set),
- A **interactive website** with two modes:
  - a **benchmark viewer** for browsing benchmarks, and
  - an **annotation workflow** originally used for data collection (collection is closed, but a test login is available).


## Environment Setup
```bash
# create conda environment
conda create -n tasmap_plus_plus python=3.8 -y
conda activate tasmap_plus_plus

# install dependencies
pip install -r requirements.txt
```



## Benchmark Website

### Benchmark Viewer

Open the benchmark selector:

* [http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/select.html?demo=true](http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/select.html?demo=true)

If you are redirected to `index.html`, open the main page first, click **Run Demo**, and then proceed:

* [http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/index.html](http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/index.html)

Next, select **Scenario → House → Room**, then click **Next** to view the benchmark.

---
### Annotation Workflow (Test Only)

Annotation collection has ended, but you can still test the annotation flow.

1. Open:

   * [http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/index.html](http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/index.html)
2. Click **Start Annotation** and log in with:

   * **Name**: `anonymous`
   * **User ID**: `20250101`

  
