//
//  Initializers.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 2/28/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import swix

public class Initializer {
    public typealias Function = ((Int, Int)?) -> matrix
    private let _fn: Function
    
    public init(_ fn: @escaping Function) {
        _fn = fn
    }
    
    public func initialize(_ shape: (Int, Int)?) -> matrix {
        return _fn(shape)
    }
}

public final class ZerosInitializer : Initializer {
    public init() {
        super.init { shape in zeros(shape!) }
    }
}

public final class GlorotInitializer : Initializer {
    public init() {
        super.init { shape in glorotn(shape!, fanIn: shape!.0, fanOut: shape!.1) }
    }
}

public final class ConstantInitializer : Initializer {
    public init(_ m: matrix) {
        super.init { _ in m }
    }
}
