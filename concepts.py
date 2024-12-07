


"""
Quantization
=============
a) symmetric               b)asymmetric

1- post training           2- during training






LLM fine tuning
=================

1- LORA (Low rank  Adaptation for LLM fine tuning ) : 
            - using matrix decomposition
            - parameter (rank)
            - low rank gives smaller model
            - very high rank gives larger model ? ---> used when fine tuning in very complex domain
            - LORA =====>>>  model_weights * ΔW  =  model_weights * A*B
                where A and B are low rank approximation for ΔW
            - "tracked weights" refer to the low-rank matrices A and B that are learned during the fine-tuning process.

Example Workflow
=================

    Pre-trained Model: Start with a pre-trained LLM with weight matrix WW.

    Initialize Low-Rank Matrices: Initialize A and B with small dimensions 
                                (e.g., if W is 1000X10001000X1000, AA might be 1000X101000X10 and B might be 10X100010X1000).

    Fine-Tuning Process: During training, only A and B are updated to minimize the loss function, effectively learning ΔW.

    Inference: During inference, the modified weights W+ΔW are used, where ΔW is the product of the learned A and B.

"""





