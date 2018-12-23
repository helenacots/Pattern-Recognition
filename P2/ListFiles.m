function files = ListFiles(directory)
f = dir(directory);
files = [];
for i=1:size(f,1)
    if f(i).isdir==0,
        if strcmp(f(i).name(end-3:end),'tiff')==1
            files = [files ; f(i)];
        end
    end
end