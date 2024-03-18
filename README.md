# CS4TE_master
 CS4TE: A Novel Coded Self-Attention and Semantic Synergy Network for Triple Extraction

## Usage

1. **Environment**
   ```shell
   conda create -n your_env_name python=3.8
   conda activate your_env_name
   cd CS4TE_master
   pip install -r requirements.txt
   ```

2. **The pre-trained BERT**

    The pre-trained BERT (bert-base-cased) will be downloaded automatically after running the code. Also, you can manually download the pre-trained BERT model and decompress it under `./pre_trained_bert`.


3. **Train the model (take WebNLG as an example)**

    run
    ```shell
    python train.py --train
    ```
    The model weights with the best performance on dev dataset will be stored in `checkpoint/WebNLG/`

4. **Evaluate on the test set (take WebNLG as an example)**

    Modify the `model_name` (line 48) to the name of the saved model, and run 
    ```shell
    python test.py
    ```
    The extracted results will be save in `result/WebNLG`.
   
## Thank
https://github.com/ssnvxia/OneRel
