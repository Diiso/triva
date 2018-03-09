function nms_edges=non_max_suppression(dI_norm,quantified_orientation,threshold)
% perform non_maximum suppression to find the edges 
% nms_edges is a boolean matrix.

[H,W]=size(dI_norm);

% find candidate edges
[candidate_edges_i,candidate_edges_j]=find(dI_norm>threshold);

nms_edges=zeros(size(dI_norm)-2); % for simplicity, this code doesn't 
% consider the borders. 

% consider all candidate edges
for k=1:length(candidate_edges_i)
    i=candidate_edges_i(k);
    j=candidate_edges_j(k);
    
    % if the candidate is on a border, do nothing
    if i==1 || j==1 || i==H || j==W
        continue
    end
    
    %% Part to complete: when is there edges for each orientation?
    if quantified_orientation(i,j)==1
    elseif quantified_orientation(i,j)==2
    elseif quantified_orientation(i,j)==3
    elseif quantified_orientation(i,j)==4
    else
        error('non_max_suppression: the input orientation matrix is not valid\n');
    end
end
        
end