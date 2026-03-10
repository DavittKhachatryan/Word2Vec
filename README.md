# Word2Vec SGNS in NumPy

Pure NumPy implementation of the Skip-Gram Word2Vec model with negative sampling.

## **Features**
 
- Subsampling of frequent words to reduce training noise and speed up learning  
- Dynamic window size for context words  
- Learning rate decay during training  
- `most_similar` function to find nearest neighbors in embedding space

## **Requirements**

- Python ≥ 3.7  
- NumPy  

Install dependencies:

```Bash
pip install -r requirements.txt
```

## **Run Training Example**

To train the Word2Vec model on `Alice.txt` and see nearest words:

```Bash
python train_example.py
```
