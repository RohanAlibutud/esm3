#||========================================================================================||
#||                                                                                        ||
#||     ╔═══╗╔═══╗╔═╗╔═╗     ╔═══╗    ╔══╗╔═╗ ╔╗ ╔═══╗╔═══╗╔═══╗╔═══╗╔═╗ ╔╗╔═══╗╔═══╗      ||
#||     ║╔══╝║╔═╗║║║╚╝║║     ║╔═╗║    ╚╣╠╝║║╚╗║║ ║╔══╝║╔══╝║╔═╗║║╔══╝║║╚╗║║║╔═╗║║╔══╝      ||
#||     ║╚══╗║╚══╗║╔╗╔╗║     ╚╝╔╝║     ║║ ║╔╗╚╝║ ║╚══╗║╚══╗║╚═╝║║╚══╗║╔╗╚╝║║║ ╚╝║╚══╗      ||
#||     ║╔══╝╚══╗║║║║║║║╔═══╗╔╗╚╗║     ║║ ║║╚╗║║ ║╔══╝║╔══╝║╔╗╔╝║╔══╝║║╚╗║║║║ ╔╗║╔══╝      ||
#||     ║╚══╗║╚═╝║║║║║║║╚═══╝║╚═╝║    ╔╣╠╗║║ ║║║╔╝╚╗  ║╚══╗║║║╚╗║╚══╗║║ ║║║║╚═╝║║╚══╗      ||
#||     ╚═══╝╚═══╝╚╝╚╝╚╝     ╚═══╝    ╚══╝╚╝ ╚═╝╚══╝  ╚═══╝╚╝╚═╝╚═══╝╚╝ ╚═╝╚═══╝╚═══╝      ||
#||                                                                                        ||                                                                             
#||========================================================================================||    

"""
VERSION 1.0
> Created 2024.07.08
> Updated 2024.07.09
"""
                                                                         

"""
╔╗   ╔══╗╔══╗ ╔═══╗╔═══╗╔═══╗╔══╗╔═══╗╔═══╗
║║   ╚╣╠╝║╔╗║ ║╔═╗║║╔═╗║║╔═╗║╚╣╠╝║╔══╝║╔═╗║
║║    ║║ ║╚╝╚╗║╚═╝║║║ ║║║╚═╝║ ║║ ║╚══╗║╚══╗
║║ ╔╗ ║║ ║╔═╗║║╔╗╔╝║╚═╝║║╔╗╔╝ ║║ ║╔══╝╚══╗║
║╚═╝║╔╣╠╗║╚═╝║║║║╚╗║╔═╗║║║║╚╗╔╣╠╗║╚══╗║╚═╝║
╚═══╝╚══╝╚═══╝╚╝╚═╝╚╝ ╚╝╚╝╚═╝╚══╝╚═══╝╚═══╝                                                                                 
"""

# MACHINE LEARNING LIBRARIES
import torch
import torch.nn.functional as F

# ESM-3 LIBRARIES
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.pretrained import ESM3_sm_open_v0, ESM3_structure_encoder_v0
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer as EsmFunctionTokenizer,
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from esm.utils.constants.esm3 import (
    SEQUENCE_MASK_TOKEN,
)
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation

# PHYLOGENETIC LIBRARIES
from Bio import SeqIO

# STATISTICAL LIBRARIES
import pandas as pd
import numpy as np

# FILEHANDLING LIBRARIES
import os
import tqdm as tqdm

"""
╔═╗╔═╗╔═══╗╔════╗╔╗ ╔╗╔═══╗╔═══╗╔═══╗
║║╚╝║║║╔══╝║╔╗╔╗║║║ ║║║╔═╗║╚╗╔╗║║╔═╗║
║╔╗╔╗║║╚══╗╚╝║║╚╝║╚═╝║║║ ║║ ║║║║║╚══╗
║║║║║║║╔══╝  ║║  ║╔═╗║║║ ║║ ║║║║╚══╗║
║║║║║║║╚══╗ ╔╝╚╗ ║║ ║║║╚═╝║╔╝╚╝║║╚═╝║
╚╝╚╝╚╝╚═══╝ ╚══╝ ╚╝ ╚╝╚═══╝╚═══╝╚═══╝
"""

# ||=========================||
# ||    INFERENCE METHODS    ||
# ||=========================||

# METHOD TO INITIALIZE MODEL
def model_init():
    model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cpu") # can also be "cuda"
    
    return model

# METHOD TO PERFORM INFERENCE ON A SEQUENCE
def infer_sequence(model, sequence):
    
    # remove gaps
    sequence = sequence.replace('-', '')
    
    # intialize model and tokenizer
    tokenizer = EsmSequenceTokenizer()
    
    # Tokenize the sequence
    tokens = tokenizer(sequence)['input_ids']
    tokens = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).cpu()  # Add batch dimension and move to CPU

    
    # same inference structure as ESM2?
    with torch.no_grad():
        output = model.forward(sequence_tokens=tokens)
        sequence_logits = output.sequence_logits    
        
        #aa_logits = sequence_logits[0, :, :20]  # select amino acid logits only
        aa_logits = sequence_logits[0, 1:-1, 4:32]
        
        token_probs = torch.softmax(aa_logits, dim=-1)
    
    # Remove the first and last token (they are likely empty placeholders or special tokens)
    #token_probs = token_probs[1:-1]
    new_token_probs = token_probs.cpu().numpy()
    
    # output as dataframe
    #df = pd.DataFrame(new_token_probs, columns=list('ACDEFGHIKLMNPQRSTVWYXBUZO.-|'))
    df = pd.DataFrame(new_token_probs, columns=list('LAGVSERTIDPKQNFYMHWCXBUZO.-|'))
    #print(np.argmax(new_token_probs, axis = -1))
    
    
    return df
    
# METHOD TO RUN INFERENCE ON AN ALIGNMENT
def infer_alignment(model, alignment_folder, alignment_name, output_folder):
    with open(f"{alignment_folder}/{alignment_name}", "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence = str(record.seq)
            species = str(record.description)
            output_df = infer_sequence(model, sequence)
            domain = alignment_name.strip("formatted_").strip("-seqs.fasta")
            output_path = f"{output_folder}/{domain}_{species}.csv"
            output_df.to_csv(output_path, index=False)


"""
╔═══╗╔═══╗╔═╗╔═╗╔═╗╔═╗╔═══╗╔═╗ ╔╗╔═══╗╔═══╗
║╔═╗║║╔═╗║║║╚╝║║║║╚╝║║║╔═╗║║║╚╗║║╚╗╔╗║║╔═╗║
║║ ╚╝║║ ║║║╔╗╔╗║║╔╗╔╗║║║ ║║║╔╗╚╝║ ║║║║║╚══╗
║║ ╔╗║║ ║║║║║║║║║║║║║║║╚═╝║║║╚╗║║ ║║║║╚══╗║
║╚═╝║║╚═╝║║║║║║║║║║║║║║╔═╗║║║ ║║║╔╝╚╝║║╚═╝║
╚═══╝╚═══╝╚╝╚╝╚╝╚╝╚╝╚╝╚╝ ╚╝╚╝ ╚═╝╚═══╝╚═══╝
""" 

# PERFORM TEST INFERENCE ON SEQUENCE
t1_active = False
if t1_active:
    test_seq = "DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPP"
    model = model_init()
    df = infer_sequence(model, test_seq)
    df.to_csv("/home/esl/ESM/ESM3/test_inference.csv", float_format = '%.10f', index = False)
    print("Finished running t1: test sequence inference")

  
# PERFORM TEST INFERENCE ON ALIGNMENT    
t2_active = True
if t2_active:
    model = model_init()
    alignment_folder = "/home/esl/ESM/Data/pf00001_alignments/test_alignments"
    for file in os.listdir(alignment_folder):
        output_folder = "/home/esl/ESM/ESM3/Results/pf1_test_ten"
        infer_alignment(model, alignment_folder, file, output_folder)
    print("Finished running t2: test alignment inference")
    
    
    
    
    
    
    
    
    
    
    
    