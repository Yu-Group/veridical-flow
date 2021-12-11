cd ../vflow
pdoc --html . --output-dir ../docs --template-dir . # need to have pip installed pdoc3
cp -rf ../docs/vflow/* ../docs/
rm -rf ../docs/vflow
cd ../docs
python style_docs.py
