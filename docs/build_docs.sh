cd ../pcs
pdoc --html . --output-dir ../docs --template-dir .
cp -rf ../docs/pcs/* ../docs/
rm -rf ../docs/pcs
cd ../docs
python3 style_docs.py
