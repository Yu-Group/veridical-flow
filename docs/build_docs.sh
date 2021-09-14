cd ../vflow
pdoc --html . --output-dir ../docs --template-dir .
cp -rf ../docs/vflow/* ../docs/
rm -rf ../docs/vflow
cd ../docs
python3 style_docs.py
