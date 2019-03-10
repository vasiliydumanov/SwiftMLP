
Pod::Spec.new do |s|

  s.name         = "SwiftMLP"
  s.version      = "0.0.1"
  s.summary      = "Swift MLP library based on Swix (https://github.com/vasiliydumanov/swix)"

  s.description  = <<-DESC
  Swift MLP library based on Swix (https://github.com/vasiliydumanov/swix)
                   DESC
  s.homepage     = "https://github.com/vasiliydumanov/swix"
  s.license      = { :type => "MIT", :file => "LICENSE" }

  s.author             = { "Vasiliy Dumanov" => "vasiliy.dumanov@gmail.com" }
  s.platform     = :ios
  s.ios.deployment_target = '12.0'
  s.source       = { :git => "https://github.com/vasiliydumanov/SwiftMLP.git", :tag => "#{s.version}" }
  s.source_files  = "SwiftMLP/*.swift", "SwiftMLP/**/*.swift"
  s.dependency 'swix', '~> 1.0.11'
  s.swift_version = "4.2"

end