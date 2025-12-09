class GradOperation:
    def __init__(self, op_name: str, backward_fn: callable, operands: list):
        self.op_name = op_name
        self.operands = operands
        self.backward_fn = backward_fn

    def backward(self, grad):
        for operand in self.operands:
            if operand.requires_grad:
                operands_without_self = self.operands.copy()
                operands_without_self.remove(operand)
                new_grad = self.backward_fn(operand, operands_without_self)
                operand.backward(new_grad * grad)

    def __str__(self):
        return "GradOperation(" + self.op_name + ", " + str(self.operands) + ")"
    
    def __repr__(self):
        return "GradOperation(" + self.op_name + ", " + repr(self.operands) + ")"
