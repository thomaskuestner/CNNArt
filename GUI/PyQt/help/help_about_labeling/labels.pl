# LaTeX2HTML 2018 (Released Feb 1, 2018)
# Associate labels original text with physical files.


$key = q/fig:create_label/;
$external_labels{$key} = "$URL/" . q|node1.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:cursor_inspector/;
$external_labels{$key} = "$URL/" . q|node5.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:edit_default_label/;
$external_labels{$key} = "$URL/" . q|node1.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:edit_label/;
$external_labels{$key} = "$URL/" . q|node1.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:select_ROI/;
$external_labels{$key} = "$URL/" . q|node5.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:set_result_color/;
$external_labels{$key} = "$URL/" . q|node3.html|; 
$noresave{$key} = "$nosave";

$key = q/fig:show_result/;
$external_labels{$key} = "$URL/" . q|node2.html|; 
$noresave{$key} = "$nosave";

1;


# LaTeX2HTML 2018 (Released Feb 1, 2018)
# labels from external_latex_labels array.


$key = q/fig:cursor_inspector/;
$external_latex_labels{$key} = q|4|; 
$noresave{$key} = "$nosave";

$key = q/fig:select_ROI/;
$external_latex_labels{$key} = q|3|; 
$noresave{$key} = "$nosave";

$key = q/fig:set_result_color/;
$external_latex_labels{$key} = q|2|; 
$noresave{$key} = "$nosave";

$key = q/fig:show_result/;
$external_latex_labels{$key} = q|1|; 
$noresave{$key} = "$nosave";

1;

