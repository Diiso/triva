function edges=canny_edges(I,sigma,t1,t2)
% t1>t2


I_blurred=gaussian_convolution(I,sigma); 
[dIx dIy dI_norm dI_orientation]=compute_gradient(I_blurred);
quantified_orientation=quantify_gradient(dI_orientation);
nms_edges_1=non_max_suppression(dI_norm,quantified_orientation,t1);
nms_edges_2=non_max_suppression(dI_norm,quantified_orientation,t2);
edges=nms_edges_1;
[H,W]=size(edges);

%% Hysteresis
[edges_to_visit_i,edges_to_visit_j]=find(nms_edges_1); % find the indices
% of the most confident edges

while ~isempty(edges_to_visit_i)
    for k=1:length(edges_to_visit_i)
        edge_i=edges_to_visit_i(k);
        edge_j=edges_to_visit_j(k);
        %% Part to complete: 
        % Where can there be a new edge? With which condition is it indeed a new edge?
        % Understand why the code will work if you just put the
        % value of 'edges(i,j)' to 1 for those new edges
        % Warning; be sure your indices don't go out of the arrays
        
        % For each edge in nms_edges_1, we check each surrounding pixel to see if it can be prolongated as an edge coming from nms_edges_2
        for i=-1:2
          for j=-1:2
            if (i==0) & (j==0)
              % We don't check the pixel we are on...
              continue
            end
            if (edge_i+i>=0) & (edge_i+i<=H) & (edge_j+j>=0) & (edge_j+j<=W)
              potential_edge_i = edge_i + i;
              potential_edge_j = edge_j + j;
              if (nms_edges_2(potential_edge_i,potential_edge_j) == 1) & (nms_edges_1(potential_edge_i,potential_edge_j) == 0)
                edges(potential_edge_i,potential_edge_j) = 1;
              end
            end
          end
        end
    end
    [edges_to_visit_i,edges_to_visit_j]=find(edges & ~nms_edges_1); % find the edges
    % that have not yet been extended
    nms_edges_1=edges;
end
