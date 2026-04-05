Notes on open weight models:

According to the paper, TurboQuant was thoroughly tested with Qwen and Mistral models. While architecturally not too different from the Qwen family, and architecture should be arbitrary in this case as we are working with quantization, we notice that Qwen 2.5 7B suffers catastrophic loss due to a few key layers having extraordinarily high max key norms. 

Reproducing TurboQuant results in Llama 3.1-8B shows up to 80% memory savings