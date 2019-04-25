# LaTeX2HTML 2018 (Released Feb 1, 2018)
# Associate labels original text with physical files.


$key = q/fig:datasets_interface/;
$external_labels{$key} = "$URL/" . q|node2.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:network_interface/;
$external_labels{$key} = "$URL/" . q|node4.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:perform_patching/;
$external_labels{$key} = "$URL/" . q|node3.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:perform_training/;
$external_labels{$key} = "$URL/" . q|node4.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:select_datasets/;
$external_labels{$key} = "$URL/" . q|node2.html|; 
$noresave{$key} = "$nosave";

1;


# LaTeX2HTML 2018 (Released Feb 1, 2018)
# labels from external_latex_labels array.


$key = q/fig:datasets_interface/;
$external_latex_labels{$key} = q|2|; 
$noresave{$key} = "$nosave";

$key = q/fig:network_interface/;
$external_latex_labels{$key} = q|5|; 
$noresave{$key} = "$nosave";

$key = q/fig:perform_patching/;
$external_latex_labels{$key} = q|3|; 
$noresave{$key} = "$nosave";

$key = q/fig:perform_training/;
$external_latex_labels{$key} = q|4|; 
$noresave{$key} = "$nosave";

$key = q/fig:select_datasets/;
$external_latex_labels{$key} = q|1|; 
$noresave{$key} = "$nosave";

1;

