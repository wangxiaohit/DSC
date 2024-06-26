# DSC
A Collaborative Service Composition Approach Considering Providers' Self-Interest and Minimal Service Sharing

**Note: This README is being updated. The code for candidate service reduction is currently under reorganization. We appreciate your patience as we improve the project's structure and documentation.**

## Information

## Installation
1. Clone the repository and install required dependencies:
    ```angular2html
    git clone https://github.com/wangxiaohit/DSC.git
    cd DSC
    pip install -r requirements.txt
    ```
2. Download pre-trained model outputs:
    - Download from: [Pre-trained Model Outputs](https://drive.google.com/file/d/11qL1oWc78010t2w6acrivRw0cPySN7R5/view?usp=drive_link)
    - After downloading, move the file to `<work_dir>/solutions/`

## Usage

### Running the Main Script

For the proposed approach:
```
python main.py HugCom Ours 0.3 20 0 5
```

For baseline approaches:
```
python main.py [APPROACH]
```
Where `[APPROACH]` can be one of: CSC, CSSA, MOPSO, SBOTI, TSDSC

### Parameters Explanation

- `HugCom`: Dataset name
- `Ours`: Method name
- `0.3`: Threshold for service availability prediction probability
- `20`: Time window
- `0`: Index of the GNN model output to use (0 for the first model output in the downloaded file)
- `5`: The top k services with the highest prediction probabilities to consider