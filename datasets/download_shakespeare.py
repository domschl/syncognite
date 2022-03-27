try:
    from ml_indie_tools.Gutenberg_Dataset import Gutenberg_Dataset
except ImportError:
    print("Please install ml_indie_tools: pip install ml_indie_tools")
    exit(1)

gd = Gutenberg_Dataset()
print("Loading Gutenberg index...")
gd.load_index()

print("Downloading Complete Works of William Shakespeare...")
shakespeare_record = gd.get_book("100C")  # 100C is the id of the shakespeare collected works

if "text" not in shakespeare_record:
    print(f"The book text is not available in the dataset {shakespeare_record}")
    exit(1)

text = shakespeare_record["text"]
if len(text) == 0:
    print(f"The book text within {shakespeare_record} is empty")
    exit(1)

with open("shakespeare.txt", "w") as f:
    f.write(text)

print("The complete works of William Shakespeare text has been saved in shakespeare.txt")
print("This can be used as a longer version of tiny_shakespeare.txt for training")

