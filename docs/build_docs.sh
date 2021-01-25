cd ../pcsp
pdoc --html . --output-dir ../docs --template-dir .
cp -rf ../docs/pcsp/* ../docs/
rm -rf ../docs/pcsp
cd ../docs
python3 style_docs.py
