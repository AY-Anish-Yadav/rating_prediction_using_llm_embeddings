from angle_emb import AnglE
import voyageai

def load_model(model_name):
    if model_name=="UAE":
        model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    if model_name=="voyageai":
        model=voyageai.Client()
    return model


# doc_vecs = model.encode([
#     'The weather is great!'
# ], normalize_embedding=True)

# # print(doc_vecs)