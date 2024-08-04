from setuptools import find_packages,setup
setup(
    name='Review-Rating',
    version='0.0.1',
    author='Anish Yadav',
    author_email='reach.anish.yadav@gmail.com',
    install_requires=["flask","jsonify","joblib","angle-emb","request","gdown","pandas","numpy","tqdm","voyageai","scikit-learn","tensorflow"],
    packages=find_packages()
)