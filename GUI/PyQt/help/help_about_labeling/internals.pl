# LaTeX2HTML 2018 (Released Feb 1, 2018)
# Associate internals original text with physical files.


$key = q/fig:create_label/;
$ref_files{$key} = "$dir".q|node1.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:edit_default_label/;
$ref_files{$key} = "$dir".q|node1.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:edit_label/;
$ref_files{$key} = "$dir".q|node1.html|; 
$noresave{$key} = "$nosave";

1;

