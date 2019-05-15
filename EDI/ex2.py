import re
import csv
import arff
from pprint import pprint

def load_html_file(filename):
    with open(filename, "r") as html_file:
        data = html_file.read()

    return data


def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    clean_text = re.sub(cleanr, '', raw_html)
    clean_comments = re.compile('(?=<!--)([\s\S]*?)-->')  # multi line comments are still visible
    text_without_comments = re.sub(clean_comments, '', clean_text)
    clean_links = re.compile('\[.*?\]')
    output_text = re.sub(clean_links, '', text_without_comments)

    return output_text


def postprocess_text(text):
    documents = text.rstrip('\r\n')
    documents = documents.replace('\n', "")
    documents = documents.replace('\t', "")
    documents = documents.replace('\r', "")
    documents = documents.split("DOKUMENT")

    return documents


filename = "travelassist_conc.html"
html_data = load_html_file(filename)

cleaned_text = clean_html(html_data)

documents_list = postprocess_text(cleaned_text)

output = [[] for _ in range(len(documents_list))]

print("@relation dokumenty")
print("@attribute tytul string")
print("@attribute zawartosc string")

print("@data")

for i, doc in enumerate(documents_list):
    output[i] = ["Dokument_{}".format(str(i+1)), doc]
    # output['Dokument ' + str(i + 1)] = doc


# pprint(output)

#
with open('crawl_csv_better.arff', mode='w') as input_file:
    input_writer = csv.writer(input_file, delimiter=',')
    for i in output:
        # print (i)
        # input_writer.writerow(i)


# with open('crawl_csv_better [SubtitleTools.com].arff', 'rb') as data_arff:
#     # lines = [x.decode('cp1252').strip() for x in data_arff.readlines()]
#     f = arff.load(data_arff)
#     # print (arff.dumps(f))

