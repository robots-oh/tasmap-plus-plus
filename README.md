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

### Quick Links
- 🏠 **Main**: [Open Main Page][main]
- ▶️ **Viewer**: [Run Benchmark Viewer][viewer]
- ✍️ **Annotation**: [Test Annotation Workflow][annot]


### Benchmark Viewer
1. Click [Run Benchmark Viewer][viewer]
2. Select **Scenario → House → Room**
3. Click **Next**

If you are redirected to [main page][main], click **View Benchmark**, and then return to the benchmark viewer.


---

### Annotation Workflow
1. Click [Test Annotation Workflow][annot]
2. Enter your name and date of birth to log in, then click **Start**.
3. 
If you are redirected to [main page][main], click **Start Annotation**, and then return to the annotation workflow.

⚠️ Annotation collection has ended, but you can still test the annotation flow.
Log in with **Test credentials**.

- **Test credentials**: `anonymous` / `20250101`


---

[main]: http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/index.html
[viewer]: http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/select.html?demo=true
[annot]: http://ec2-43-201-242-118.ap-northeast-2.compute.amazonaws.com/index.html#login

  
