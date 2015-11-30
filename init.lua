require 'libhungarian'

function torch.hungarian(input)
   local output = torch.Tensor()
   local indices = torch.LongTensor()
   output:resize(input:size()):fill(0)
   indices:resize(input:size()):fill(0)
   input.torch.hungarian(input, output, indices)
   return output, indices
end
