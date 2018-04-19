%% ªÒ»°…„œÒÕ∑
%% 
try
    camera = webcam;
catch
    clear camera
    camera = webcam;
end

keepRolling = 1;

im = snapshot(camera);
[x, y, z] = size(im);

% -------------------------------------------------------------------------
%% net

net=load('.\NYU\models\NYU_ResNet-UpProj.mat');
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
out = net.getVarIndex('prediction');

% -------------------------------------------------------------------------
%% Options
netOpts.gpu = 0;     % set to true to enable GPU support
netOpts.plot = true;    % set to true to visualize the predictions during inference


while keepRolling
    images = snapshot(camera);
%     im_orj = im; 
%     im=imresize(im, [460,345]);
    images = imresize(images, net.meta.normalization.imageSize(1:2));
    groundTruth = [];
    
    % Get output size for initialization
    varSizes = net.getVarSizes({'data', net.meta.normalization.imageSize});  % get variable sizes
    pred = zeros(varSizes{out}(1), varSizes{out}(2), varSizes{out}(3), size(images, 4));    % initiliaze 

    if netOpts.plot, figure(1); end
    
    im = single(images);
    if netOpts.gpu
        im = gpuArray(im);
    end
    
    % run the CNN
    inputs = {'data', im};
    net.eval(inputs) ;
    
    % obtain prediction
    pred  = gather(net.vars(out).value);
    
    if netOpts.plot
        colormap jet
        
        subplot(1,2,1), imagesc(uint8(images)), title('RGB Input'), axis off
        subplot(1,2,2), imagesc(pred), title('Depth Prediction'), axis off
        
        drawnow;
    end
    % Get predictions
%     predictions = DepthMapPrediction(images, net, netOpts);
    
    
end
