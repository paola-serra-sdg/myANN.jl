using Flux: onecold

# Accuracy metric
# Todo: modify to work with more than two labels

function accuracy(y_true::Any, y_pred::Any)
    s = 0
    a = onecold(y_true, 0:(size(y_true)[1]-1))
    b = onecold(y_pred, 0:(size(y_pred)[1]-1))
    for i in range(1, size(a)[1])
        if a[i] == b[i]
            s = s+1
        end
    end
    return s/(size(a)[1])
end
