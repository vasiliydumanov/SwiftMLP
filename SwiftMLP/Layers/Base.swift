//
//  Layer.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 2/28/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios

public class Layer {
    public func forward(_ input: matrix) -> matrix {
        preconditionFailure("Subclass must override this method.")
    }
    public func backprop(_ outputGrad: matrix) -> matrix {
        preconditionFailure("Subclass must override this method.")
    }
}

public typealias SerializedLayerData = [String: matrix]

public class LayerWithParameters : Layer {
    public var states: [[matrix]]?
    public var gradients: [[matrix]] = []
    
    public func apply(gradients: [matrix]) {
        preconditionFailure("Subclass must override this method.")
    }
    
    public func resetGradients() {
        gradients = []
    }
    
    public func encode() -> SerializedLayerData {
        preconditionFailure("Subclass must override this method.")
    }
    
    public func decode(_ data: SerializedLayerData) {
        preconditionFailure("Subclass must override this method.")
    }
}

