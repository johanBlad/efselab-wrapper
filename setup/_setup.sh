echo 'Build lemmatizer...'
python setup.py build_ext --inplace

# Download and build models
echo 'Unpack src files...'
if [ ! -e swe-pipeline-ud2.tar.gz ]; then
    wget http://mumin.ling.su.se/projects/efselab/swe-pipeline-ud2.tar.gz
fi
tar xvzf swe-pipeline-ud2.tar.gz

echo 'Build and train suc and suc-ne models...'
python build_suc.py --skip-generate --python --n-train-fields 2
python build_suc_ne.py --skip-generate --python --n-train-fields 4


# Build and train the SUC-to-UD conversion model
echo 'Build and train udt-suc model...'
python build_udt_suc_sv.py --python --beam-size 1 --n-train-fields 4
./udt_suc_sv train \
    data/sv-ud-train.tab data/sv-ud-dev.tab swe-pipeline/suc-ud.bin

# Copy necessary binaries to bin folder, for usage in the pipeline
echo 'Copy binaries to efselabwrapper/bin...'
mkdir ../efselabwrapper/bin/
cp swe-pipeline/suc-ne.bin ../efselabwrapper/bin/
cp swe-pipeline/suc-saldo.lemmas ../efselabwrapper/bin/
cp swe-pipeline/suc-ud.bin ../efselabwrapper/bin/
cp swe-pipeline/suc.bin ../efselabwrapper/bin/

cp *.so ../efselabwrapper

echo 'Finished!'