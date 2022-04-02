try:
    from ml_indie_tools.Gutenberg_Dataset import Gutenberg_Dataset
except ImportError:
    print("Please install ml-indie-tools: pip install ml-indie-tools, see https://github.com/domschl/ml-indie-tools")
    exit(1)

gd = Gutenberg_Dataset()
print("Loading Gutenberg index...")
gd.load_index()

search_spec= {"author": ["Emilie Brontë","Jane Austen", "Virginia Woolf"], "language": ["english"]}

book_list=gd.search(search_spec)
book_cnt = len(book_list)
print(f"{book_cnt} matching books found with search {search_spec}.")
if book_cnt<40:
    # Note: please verify that book_cnt is 'reasonable'. If you plan to use a large number of texts, 
    # consider [mirroring Gutenberg](https://github.com/domschl/ml-indie-tools#working-with-a-local-mirror-of-project-gutenberg)
    book_list = gd.insert_book_texts(book_list, download_count_limit=book_cnt)  
else:
    print("Please verify your book_list, a large number of books is scheduled for download. ABORTED.")
    exit(1)

text = ""
for book in book_list:
    print(f"Adding {book['title']} by {book['author']}")
    text += book["text"]

with open("women_writers.txt", "w") as f:
    f.write(text)

print("Content of all books merged into women_writers.txt")


