from langchain_community.llms import CTransformers


def initialize_llm():
    model_path = "model\llama-2-7b-chat.ggmlv3.q4_0.bin"
    # providce relative path wrt root of the working directory
    llm = CTransformers(model= model_path,
                        model_type='llama',
                        config={'max_new_tokens': 725,
                                'temperature': 0.5,
                                "context_length": 5000}
                                )
    return llm


if __name__ == '__main__':
    llm = initialize_llm()
    print(llm.invoke("what is Large language model?"))