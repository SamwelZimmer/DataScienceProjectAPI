# # imports
# import torch
# from facenet_pytorch import InceptionResnetV1, MTCNN
# from PIL import Image
# import requests
# import pinecone

# # constants
# PINECONE_API_KEY = "b4637963-b62e-4063-9e14-1940d9f7f105"
# PINECONE_ENVIRONMENT = "us-west4-gcp-free"


# def get_image(url):
#     img = Image.open(requests.get(url, stream=True).raw)
#     return img


# def find_similar_faces(face, top_k=10):
#     # pass the image through the embedding pipeline
#     emb = facenet.encode([face])
#     # query pinecone with the face embedding
#     result = index.query(emb[0], top_k=6, include_metadata=True)
#     # extract metadata from the search results and display results
#     r = [x["metadata"] for x in result["matches"]]
#     return r


# class FacenetEmbedder:
#     def __init__(self):
#         # set device to use GPU if available
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#         # initialize MTCNN model
#         self.mtcnn = MTCNN(device=self.device)

#         # initialize VGGFace2 model
#         self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()

#     def detect_face(self, batch):
#         # get coordinates of the face
#         faces = self.mtcnn.detect(batch)
#         return faces

#     def encode(self, batch):
#         # pass the batch of images directly through mtcnn model
#         face_batch = self.mtcnn(batch)

#         # remove any images that does not contain a face
#         face_batch = [i for i in face_batch if i is not None]

#         # concatenate face batch to form a single tensor
#         aligned = torch.stack(face_batch)

#         # if using gpu move the input batch to gpu
#         if self.device.type == "cuda":
#             aligned = aligned.to(self.device)

#         # generate embedding
#         embeddings = self.resnet(aligned).detach().cpu()
#         return embeddings.tolist()


# # initialize the embedding pipeline
# facenet = FacenetEmbedder()

# # connect to pinecone environment
# pinecone.init(
#     api_key=PINECONE_API_KEY,
#     environment=PINECONE_ENVIRONMENT
# )

# # arbitrary vector index name
# index_name = "tmdb-people"

# # check if the tmdb-people index exists
# if index_name not in pinecone.list_indexes():
#     # create the index if it does not exist
#     pinecone.create_index(
#         index_name,
#         dimension=512,
#         metric="cosine"
#     )

# # connect to tmdb-people index we created
# index = pinecone.GRPCIndex(index_name)

# # # we'll use default for now
# # url = "https://live.staticflickr.com/7442/9509564504_21d2dc42e1_z.jpg"
# # # load the image as PIL object from url
# # celeb = get_image(url)
# # find_similar_faces(celeb)


# def get_similar_faces(url):
#     celeb = get_image(url)

#     # Assuming img is your input image
#     if celeb.mode == 'RGBA':
#         celeb = celeb.convert('RGB')

#     similar_faces = find_similar_faces(celeb)
#     return similar_faces
