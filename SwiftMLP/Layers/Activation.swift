//
//  Activation.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 2/28/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios

public class Relu : Layer {
    private var _input: matrix!
    
    public override init() { super.init() }
    
    public override func forward(_ input: matrix) -> matrix {
        _input = input
        return apply_function_flat({ $0 > 0 ? $0 : 0 }, x: input)
    }
    
    public override func backprop(_ outputGrad: matrix) -> matrix {
        return apply_function_flat({ $0 > 0 ? 1 : 0 }, x: _input) * outputGrad
    }
}

public class Softmax : Layer {
    private var _output: matrix!
    
    public override init() { super.init() }
    
    public override func forward(_ input: matrix) -> matrix {
        let expInput = apply_function({ exp($0) }, x: input)
        _output = expInput / sum(expInput, axis: 1).reshape((input.shape.0, 1))
        return _output
    }
    
    public override func backprop(_ outputGrad: matrix) -> matrix {
        preconditionFailure("This function must not be called.")
    }
}
