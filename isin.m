function in = isin(name,namelist)
in = false;

for n = 1:length(namelist)
    nm = cell2mat(namelist(n));
    in = in | strcmp(name, nm);
    if in
        break
    end
end

end

