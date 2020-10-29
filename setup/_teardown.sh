echo 'Delete .c extensions...'
rm -rf pysuc.c
rm -rf pysuc_ne.c
rm -rf pyudt_suc_sv.c
rm -rf udt_suc_sv.c

echo 'Delete binaries and shared objects...'
rm -rf udt_suc_sv
rm -rf *.so
rm -rf build