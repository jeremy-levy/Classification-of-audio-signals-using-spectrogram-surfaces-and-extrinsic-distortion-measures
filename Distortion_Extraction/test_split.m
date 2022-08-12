sig = 1:100;
len_sig = size(sig, 2);
index_base = 1;
offset = round(len_sig/5);

while index_base+offset < len_sig
    splitted_sig = sig(index_base: index_base+offset)
    index_base = index_base + offset;
end