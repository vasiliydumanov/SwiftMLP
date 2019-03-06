//
//  Layer.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 2/28/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix_ios

open class Layer {
    public init() {}
    
    open func forward(_ input: matrix) -> matrix {
        preconditionFailure("Subclass must override this method.")
    }
    open func backprop(_ outputGrad: matrix) -> matrix {
        preconditionFailure("Subclass must override this method.")
    }
}

public typealias SerializedLayerData = [String: matrix]

open class LayerWithParameters : Layer {
    public var states: [[matrix]] = []
    public var gradients: [[matrix]] = []
    
    public override init() {
        super.init()
    }
    
    open func apply(gradients: [matrix]) {
        preconditionFailure("Subclass must override this method.")
    }
    
    public func resetGradients() {
        gradients = []
    }
    
    public func resetStates() {
        states = []
    }
    
    open func encode() -> SerializedLayerData {
        preconditionFailure("Subclass must override this method.")
    }
    
    open func decode(_ data: SerializedLayerData) {
        preconditionFailure("Subclass must override this method.")
    }
}

