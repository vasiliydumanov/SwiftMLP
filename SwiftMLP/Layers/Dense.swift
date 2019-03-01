//
//  Dense.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 2/28/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios

public class Dense : LayerWithParameters {
    public let units: Int
    
    private let _wInit: Initializer
    private let _bInit: Initializer
    private var _w: matrix!
    private var _b: matrix!
    private var _input: matrix!
    
    public init(units: Int, weightsInitializer: Initializer = GlorotInitializer(), biasInitializer: Initializer = ZerosInitializer()) {
        self.units = units
        _wInit = weightsInitializer
        _bInit = biasInitializer
    }
    
    public override func forward(_ input: matrix) -> matrix {
        _input = input
        gradients = []
        if _w == nil || _b == nil {
            let inputSize = input.shape.1
            _w = _wInit.initialize((inputSize, units))
            _b = _bInit.initialize((1, units))
        }
        let z = input.dot(_w) + _b
        return z
    }
    
    public override func backprop(_ outputGrad: matrix) -> matrix {
        let inputWShaped = `repeat`(_input.flat, N: _w.shape.1, axis: 1).reshape(_w.shape)
        let wGrad = inputWShaped * outputGrad
        let bGrad = outputGrad
        gradients = [wGrad, bGrad]
        print("wGrad: \(wGrad)")
        print("bGrad: \(bGrad)")
        return sum(_w * outputGrad, axis: 1).reshape((1, _w.shape.0))
    }
    
    public override func apply(gradients: [matrix]) {
        _w = _w - gradients[0]
        _b = _b - gradients[1]
    }
    
    public override func encode() -> SerializedLayerData {
        return ["w": _w, "b": _b]
    }
    
    public override func decode(_ data: SerializedLayerData) {
        _w = data["w"]
        _b = data["b"]
    }
}
