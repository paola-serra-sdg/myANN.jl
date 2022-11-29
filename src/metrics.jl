using Flux: onecold

# Accuracy

function accuracy(y_true::Any, y_pred::Any)
    a = onecold(y_true, 0:(size(y_true)[1]-1))
    b = onecold(y_pred, 0:(size(y_pred)[1]-1))
    l = size(a)[1]
    return sum(a .== b) / l
end
