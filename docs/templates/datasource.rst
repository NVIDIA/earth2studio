:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{module}}

.. autoclass:: {{ objname }}

   {% block methods %}
    .. automethod:: __call__
    .. automethod:: available
   {% endblock %}

.. raw:: html

    <div class="clearer"></div>
