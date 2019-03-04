//
//  Logging.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/4/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation



public final class Logging : Callback {
    public typealias OnEpochEnd = (Int, Log) -> ()
    
    private let _completion: OnEpochEnd
    
    public override var priority: Priority {
        return .end
    }
    
    public init(onEpochEnd: @escaping OnEpochEnd) {
        _completion = onEpochEnd
        super.init()
    }
    
    public override func onEpochEnd(epoch: Int, log: inout Log) -> Bool {
        _completion(epoch, log)
        return true
    }
}
