
function imdsTrimmed = Separator(imdsTestA, train, dataset)


files = cell(train.Files);

str1 = dataset;
str2 = 'Test';

files = strrep(files,str1,str2);

subd = [];
subd2 = 0;

for(i = 1:length(files))
    for(k = 1:length(imdsTestA.Files))
        if(isequal(files{i}, imdsTestA.Files{k}))
            subd2 = 1;
            continue
            %subd2 = isequal(files{i}, imdsTestA.Files{k}) + subd2; 
        end
    end
       if(subd2 == 0)
            subd = [subd, i];
       end  
        
     subd2 = 0;

end

imdsTrimmed = subset(train, subd);

end