//
//  EarlyStopping.swift
//  SwiftMLP
//
//  Created by Vasiliy Dumanov on 3/4/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation


public class EarlyStopping : Callback {
    public let monitor: LogKey
    public let minDelta: Double
    public let patience: Int
    
    private var _patienceCounter: Int = 0
    private var _bestParams: SerializedModelData?
    private var _bestEpoch: Int?
    private var _bestMonitored: Double = .greatestFiniteMagnitude
    
    public override var priority: Priority {
        return .begin
    }
    
    public init(monitor: LogKey = .valLoss, minDelta: Double = 0, patience: Int = 0) {
        self.monitor = monitor
        self.minDelta = minDelta
        self.patience = patience
    }
    
    public override func onEpochEnd(epoch: Int, log: inout Log) -> Bool {
        guard let monitored = log[monitor] as? Double else {
            fatalError("no value for key \"monitor\" in `log`")
        }
        if monitored < _bestMonitored - minDelta {
            _patienceCounter = 0
            _bestMonitored = monitored
            _bestEpoch = epoch
            _bestParams = _model.serialize()
        } else {
            if _patienceCounter == patience {
                log[.esEpoch] = _bestEpoch!
                log[.esMonitor] = monitor
                log[.esBestMonitored] = _bestMonitored
                log[.esLogStr] = "Early stopping: Restoring parameters from epoch \(_bestEpoch!), \(monitor.rawValue) = \(_bestMonitored)"
                print(log[.esLogStr] as! String)
                _model.restore(from: _bestParams!)
                return false
            }
            _patienceCounter += 1
        }
        return true
    }
    
    public override func onTrainEnd(log: inout Log) {
        _bestMonitored = .greatestFiniteMagnitude
        _bestParams = nil
        _bestEpoch = nil
        _patienceCounter = 0
    }
}

public extension LogKey {
    public static let esEpoch = LogKey(rawValue: "es_epoch")
    public static let esMonitor = LogKey(rawValue: "es_monitor")
    public static let esBestMonitored = LogKey(rawValue: "es_best_monitored")
    public static let esLogStr = LogKey(rawValue: "es_log_str")
}
