# Numo::NArray - New NArray class library for Ruby/Numo (NUmerical MOdule)

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/ruby-numo/narray)
[![Build Status](https://travis-ci.org/ruby-numo/narray.svg?branch=master)](https://travis-ci.org/ruby-numo/narray)

[GitHub](https://github.com/ruby-numo/narray)
 | [RubyGems](https://rubygems.org/gems/numo-narray)

Numo::NArray is an Numerical N-dimensional Array class
for fast processing and easy manipulation of multi-dimensional numerical data,
similar to numpy.ndaray.
This project is a successor to [Ruby/NArray](http://masa16.github.io/narray/).

under development

## Related Projects
* [Numo::Linalg](https://github.com/ruby-numo/linalg) - Linear Algebra library with [LAPACK](http://www.netlib.org/lapack/).
* [Numo::GSL](https://github.com/ruby-numo/gsl) - Ruby interface for [GSL (GNU Scientific Library)](http://www.gnu.org/software/gsl/).
* [Numo::FFTE](https://github.com/ruby-numo/ffte) - Ruby interface for [FFTE (A Fast Fourier Transform library with radix-2,3,5)](http://www.ffte.jp/).
* [Numo::Gnuplot](https://github.com/ruby-numo/gnuplot) - Simple and easy-to-use Gnuplot interface.

## Installation
### Ubuntu, Debian
```shell
apt install -y git ruby gcc ruby-dev rake make
git clone git://github.com/ruby-numo/narray
cd narray
gem build numo-narray.gemspec
gem install numo-narray-0.9.0.3.gem
```

## Quick start
An example
```ruby
[1] pry(main)> require "numo/narray"
=> true
[2] pry(main)> a = Numo::DFloat.new(3,5).seq
=> Numo::DFloat#shape=[3,5]
[[0, 1, 2, 3, 4],
 [5, 6, 7, 8, 9],
 [10, 11, 12, 13, 14]]
[3] pry(main)> a.shape
=> [3, 5]
[4] pry(main)> a.ndim
=> 2
[5] pry(main)> a.class
=> Numo::DFloat
[6] pry(main)> a.size
=> 15
```
For more examples, check out this [narray version of 100 numpy exercises](https://github.com/ruby-numo/narray/wiki/100-narray-exercises) (and the [IRuby Notebook](https://github.com/ruby-numo/narray/blob/master/100-narray-exercises.ipynb)).

## Documentation

All documents are primitive.

* [Numo::NArray API Doc](http://ruby-numo.github.io/narray/narray/frames.html)
* [Numo::NArray概要](https://github.com/ruby-numo/narray/wiki/Numo::NArray%E6%A6%82%E8%A6%81) (in Japanese)
* [Numo::NArray vs numpy](https://github.com/ruby-numo/narray/wiki/Numo-vs-numpy)

## Running RSpec

(in advance, install gem with --development option)

  ```shell
$ "${HOME}/.gem/ruby/2.?/bin/rspec" "${HOME}/.gem/ruby/2.4/gems/numo-narray-0.9.?.?/spec/bit_spec.rb"
$ "${HOME}/.gem/ruby/2.?/bin/rspec" "${HOME}/.gem/ruby/2.4/gems/numo-narray-0.9.?.?/spec/narray_spec.rb"
```

## YARD documents generation

(in advance, install yard gem)

  ```shell
$ cd "${HOME}/.gem/ruby/2.?/gems/numo-narray-0.9.?.?/ext/numo/narray"
$ make doc
yard doc *.c types/*.c
...
```
